/*import { getUser } from '@/auth/getUser';
import { showError } from '@/commons/notifications';
import { ENV } from '@/shared/projectEnvVariables';*/
// import { useUserPrefsStore } from '@/store';
import Axios, { AxiosError, type AxiosRequestConfig } from "axios";
import { useAuthStore } from "../../authStore.ts";

export const AXIOS_INSTANCE = Axios.create({
  //@ts-ignore
   baseURL: import.meta.env.VITE_BACKEND_API || "http://localhost:3000",
  // paramsSerializer: {
  //   // Important to use qs instead of the default URLSearchParams
  //   serialize: (params) => {
  //     return qs.stringify(params, { arrayFormat: 'comma' });
  //   },
  // },
});

export const customInstance = <T>(
  config: AxiosRequestConfig,
  options?: AxiosRequestConfig,
): Promise<T> => {
  const source = Axios.CancelToken.source();

  const promise = AXIOS_INSTANCE({
    ...config,
    ...options,
    cancelToken: source.token,
  }).then(({ data }) => data);

  // @ts-ignore
  promise.cancel = () => {
    source.cancel('Query was cancelled');
  };

  return promise;
};

AXIOS_INSTANCE.interceptors.request.use(
  (config) => {
    // const token = getUser()?.access_token;
    // const userPrefs = useUserPrefsStore.getState();
    //
    // config.headers.Authorization = token ? `Bearer ${token}` : '';
    // if (userPrefs.language) {
    //   config.headers['Accept-Language'] = userPrefs.language;
    // }

    return config;
  },
  (error) => Promise.reject(error),
);

AXIOS_INSTANCE.interceptors.request.use((config) => {
  const auth = useAuthStore.getState();
  const token =  auth.token;
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});


AXIOS_INSTANCE.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (error.response?.status == 401) {
      // showError('Seja je potekla.');
      console.log('session expired, relogging');
    }

    // if (error.response?.status == 403) {
    //   showError(t('notifications.403.message'));
    // }

    throw error;
  },
);

// In some case with react-query and swr you want to be able to override the return error type so you can also do it here like this
export type ErrorType<Error> = AxiosError<Error>;
// function showError(arg0: string) {
//   throw new Error('Function not implemented.');
// }
